import {
  Box,
  Button,
  Grid,
  Heading,
  Image,
  Input,
  SimpleGrid,
  Text,
  VStack,
} from "@chakra-ui/react"
import { createFileRoute } from "@tanstack/react-router"
import { useState } from "react"

type EvaluationResults = {
  diagnosis: string
  confidence: number
  recommendations: string
}

export const Route = createFileRoute("/evaluate")({
  component: ImageEvaluation,
  beforeLoad: async () => {
    // TODO: handle unauthorized here
  },
})

function ImageEvaluation() {
  const [images, setImages] = useState<(string | null)[]>(Array(4).fill(null))
  const [isEvaluating, setIsEvaluating] = useState(false)
  const [results, setResults] = useState<EvaluationResults | null>(null)

  const handleImageUpload = (
    e: React.ChangeEvent<HTMLInputElement>,
    index: number,
  ) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        const newImages = [...images]
        newImages[index] = reader.result as string
        setImages(newImages)
      }
      reader.readAsDataURL(file)
    }
  }

  const removeImage = (index: number) => {
    const newImages = [...images]
    newImages[index] = null
    setImages(newImages)
  }

  const handleEvaluate = async () => {
    setIsEvaluating(true)
    setTimeout(() => {
      setResults({
        diagnosis: "Normal",
        confidence: 0.95,
        recommendations: "Regular follow-up recommended",
      })
      setIsEvaluating(false)
    }, 2000)
  }

  const allImagesUploaded = images.every((img) => img !== null)

  return (
    <Box p={8}>
      <Grid templateColumns={{ base: "1fr", lg: "2fr 1fr" }} gap={8}>
        {/* Main Upload Section */}
        <Box>
          <Heading size="3xl" mb={4}>
            Mammography Analyzer
          </Heading>
          <Box mt={12}>
            <SimpleGrid columns={{ base: 1, md: 2 }} gap={12}>
              <ImageUploadColumn
                title="Left"
                mloIndex={0}
                ccIndex={2}
                images={images}
                onUpload={handleImageUpload}
                onRemove={removeImage}
              />
              <ImageUploadColumn
                title="Right"
                mloIndex={1}
                ccIndex={3}
                images={images}
                onUpload={handleImageUpload}
                onRemove={removeImage}
              />
            </SimpleGrid>
          </Box>
        </Box>

        {/* Right Sidebar */}
        <VStack align="stretch" gap={6} mt={16}>
          <Box p={4} borderWidth="1px" rounded="xl" bg="white" boxShadow="sm">
            <Heading size="md" mb={2}>
              Ready to analyze
            </Heading>
            <Button
              colorScheme="brand"
              onClick={handleEvaluate}
              loading={isEvaluating}
              disabled={!allImagesUploaded}
              w="full"
            >
              Analyze
            </Button>
          </Box>

          <Box p={4} borderWidth="1px" rounded="xl" bg="white" boxShadow="sm">
            <Heading size="md" mb={2}>
              Patient Information
            </Heading>
            <Text color="gray.500">[Form or metadata goes here]</Text>
          </Box>

          {results && (
            <Box p={4} borderWidth="1px" rounded="xl" bg="white" boxShadow="sm">
              <Heading size="md" mb={2}>
                Analysis Results
              </Heading>
              <Text>
                <strong>Diagnosis:</strong> {results.diagnosis}
              </Text>
              <Text>
                <strong>Confidence:</strong>{" "}
                {(results.confidence * 100).toFixed(2)}%
              </Text>
              <Text>
                <strong>Recommendations:</strong> {results.recommendations}
              </Text>
            </Box>
          )}
        </VStack>
      </Grid>
    </Box>
  )
}

function ImageUploadColumn({
  title,
  mloIndex,
  ccIndex,
  images,
  onUpload,
  onRemove,
}: {
  title: string
  mloIndex: number
  ccIndex: number
  images: (string | null)[]
  onUpload: (e: React.ChangeEvent<HTMLInputElement>, i: number) => void
  onRemove: (i: number) => void
}) {
  return (
    <Box>
      <Heading size="xl" mb={4} color="brand.700">
        {title}
      </Heading>
      <VStack align="stretch" gap={8}>
        <ImageUploadBox
          label={`${title} Mediolateral Oblique (MLO)`}
          index={mloIndex}
          image={images[mloIndex]}
          onUpload={onUpload}
          onRemove={onRemove}
        />
        <ImageUploadBox
          label={`${title} Craniocaudal (CC)`}
          index={ccIndex}
          image={images[ccIndex]}
          onUpload={onUpload}
          onRemove={onRemove}
        />
      </VStack>
    </Box>
  )
}

function ImageUploadBox({
  label,
  index,
  image,
  onUpload,
  onRemove,
}: {
  label: string
  index: number
  image: string | null
  onUpload: (e: React.ChangeEvent<HTMLInputElement>, i: number) => void
  onRemove: (i: number) => void
}) {
  return (
    <Box>
      {image ? (
        <Box
          position="relative"
          border="1px solid"
          borderColor="gray.200"
          rounded="md"
          overflow="hidden"
          transition="all 0.2s"
          _hover={{
            borderColor: "brand.300",
            boxShadow: "0 0 0 2px var(--chakra-colors-brand-100)",
            transform: "scale(1.01)"
          }}
        >
          <Text 
            position="absolute" 
            top={2} 
            left={2} 
            bg="white" 
            px={2} 
            py={1} 
            rounded="md"
            fontSize="sm"
            fontWeight="medium"
            zIndex={1}
          >
            {label}
          </Text>
          <Image src={image} alt={label} w="full" />
          <Button
            position="absolute"
            top={1}
            right={1}
            size="xs"
            onClick={() => onRemove(index)}
            colorScheme="red"
          >
            ×
          </Button>
        </Box>
      ) : (
        <Box
          as="label"
          border="2px dashed"
          borderColor="brand.200"
          rounded="xl"
          h="180px"
          cursor="pointer"
          display="flex"
          flexDirection="column"
          alignItems="center"
          justifyContent="center"
          transition="all 0.2s"
          _hover={{ 
            bg: "brand.50",
            borderColor: "brand.300",
            boxShadow: "0 0 0 2px var(--chakra-colors-brand-100)",
            transform: "scale(1.01)"
          }}
          position="relative"
        >
          <Text 
            position="absolute"
            top={2}
            left={2}
            fontSize="sm"
            fontWeight="medium"
          >
            {label}
          </Text>
          <Input
            type="file"
            accept="image/*"
            display="none"
            onChange={(e) => onUpload(e, index)}
          />
          <Text color="gray.400" fontSize="2xl">
            +
          </Text>
        </Box>
      )}
    </Box>
  )
}
