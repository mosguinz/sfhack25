import { Box, Button, Flex, Spacer, Text } from "@chakra-ui/react"
import { useState } from "react"

const HeaderNavbar = () => {
  const [mode, setMode] = useState("mammography")

  return (
    <Flex
      as="nav"
      bg="teal.500"
      color="white"
      padding="1rem"
      align="center"
      justify="space-between"
    >
      <Text fontSize="xl" fontWeight="bold">
        UP TO
      </Text>

      <Spacer />

      <Box>
        <Button
          colorScheme={mode === "mammography" ? "blue" : "gray"}
          variant="solid"
          mr="2"
          onClick={() => setMode("mammography")}
        >
          Mammography
        </Button>
        <Button
          colorScheme={mode === "xray" ? "blue" : "gray"}
          variant="solid"
          onClick={() => setMode("xray")}
        >
          X-ray
        </Button>
      </Box>
    </Flex>
  )
}

export default HeaderNavbar
